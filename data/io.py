import requests


def load_file(url, filename):
    print('Downloading', url, 'to', filename)
    r = requests.get(url)
    if r.status_code != 200:
        print('Failed to download', url)
        return
    with open(filename, 'wb') as f:
        f.write(r.content)
