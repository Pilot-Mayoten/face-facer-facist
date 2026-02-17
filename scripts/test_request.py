import requests
urls = [
 'https://raw.githubusercontent.com/nndl/100k-faces/master/samples/sample_0000.jpg',
 'https://source.unsplash.com/random/512x512/?portrait,face',
 'https://thispersondoesnotexist.com/image'
]
for u in urls:
    try:
        r = requests.get(u, timeout=15)
        print(u, r.status_code, len(r.content))
    except Exception as e:
        print(u, 'ERROR', e)
