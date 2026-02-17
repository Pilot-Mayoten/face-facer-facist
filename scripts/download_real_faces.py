import requests
import time
from pathlib import Path

OUT = Path("data/faces")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    # 100k-faces samples (raw GitHub)
    "https://raw.githubusercontent.com/nndl/100k-faces/master/samples/sample_{:04d}.jpg",
    # Unsplash random portrait (redirects to an image)
    "https://source.unsplash.com/random/512x512/?portrait,face",
    # ThisPersonDoesNotExist fallback
    "https://thispersondoesnotexist.com/image",
]


def download(num=20, delay=1):
    success = 0
    i = 0

    while success < num and i < num * 6:
        idx = success
        tried = False

        # try 100k-faces first for variety
        if success < 100:
            url = SOURCES[0].format(success)
            tried = True
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 5000:
                    path = OUT / f"face_{success+1:04d}.jpg"
                    with open(path, 'wb') as f:
                        f.write(r.content)
                    success += 1
                    print(f"Downloaded {path} from 100k-faces ({success}/{num})")
                    time.sleep(delay)
                    continue
            except Exception as e:
                pass

        # try Unsplash
        try:
            r = requests.get(SOURCES[1], timeout=15)
            if r.status_code == 200 and len(r.content) > 5000:
                path = OUT / f"face_{success+1:04d}.jpg"
                with open(path, 'wb') as f:
                    f.write(r.content)
                success += 1
                print(f"Downloaded {path} from Unsplash ({success}/{num})")
                time.sleep(delay)
                continue
        except Exception:
            pass

        # try thispersondoesnotexist
        try:
            r = requests.get(SOURCES[2], timeout=15)
            if r.status_code == 200 and len(r.content) > 5000:
                path = OUT / f"face_{success+1:04d}.jpg"
                with open(path, 'wb') as f:
                    f.write(r.content)
                success += 1
                print(f"Downloaded {path} from thispersondoesnotexist ({success}/{num})")
                time.sleep(delay)
                continue
        except Exception:
            pass

        i += 1
        time.sleep(0.5)

    print(f"Finished: downloaded {success}/{num} images to {OUT}")


if __name__ == '__main__':
    import sys
    n = 20
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except:
            pass
    download(num=n, delay=1)
