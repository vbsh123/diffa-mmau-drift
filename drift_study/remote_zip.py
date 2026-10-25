"""Seekable HTTP range reader. Refuses servers that ignore Range requests."""
import io
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, timeout=60):
        super().__init__()
        self.url, self.timeout, self.position = url, timeout, 0
        self.bytes_received = 0
        self.size = None
        self._fetch(0, 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset if whence == 2 else -1
        if position < 0:
            raise ValueError('Invalid seek')
        self.position = position
        return position

    def _fetch(self, start, count):
        end = start+count-1
        # Distinguish ranges in CDN cache keys; never reuse an expired signed URL.
        separator = '&' if '?' in self.url else '?'
        url = self.url+separator+urlencode(dict(download='true', range_start=start, range_end=end))
        for attempt in range(4):
            try:
                request = Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urlopen(request, timeout=self.timeout) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match:
                        raise ValueError('Server ignored HTTP Range; refusing full archive download')
                    first, last, total = map(int, match.groups())
                    if first != start or last != end or (self.size is not None and total != self.size):
                        raise ValueError('Inconsistent HTTP Content-Range')
                    self.size = total
                    data = response.read(count+1)
                    if len(data) != count:
                        raise IOError('Incomplete HTTP range')
                    self.bytes_received += len(data)
                    return data
            except (OSError, IOError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    def read(self, size=-1):
        count = max(0, self.size-self.position) if size < 0 else min(size, max(0, self.size-self.position))
        if not count:
            return b''
        result = self._fetch(self.position, count)
        self.position += len(result)
        return result
