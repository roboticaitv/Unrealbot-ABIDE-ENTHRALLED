import urllib.request
import os

url = 'http://127.0.0.1:8080/stream'
try:
    stream = urllib.request.urlopen(url, timeout=5)
    bytes_data = b''
    for _ in range(500):  # read chunks until we find a full JPEG frame
        chunk = stream.read(4096)
        if not chunk:
            break
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1 and b > a:
            jpg = bytes_data[a:b+2]
            dest_dir = '/home/pi/.gemini/antigravity-ide/brain/ae967698-017f-4a60-ae1b-f4de0b10b552'
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, 'captured_frame.jpg'), 'wb') as f:
                f.write(jpg)
            print("Capture success")
            break
except Exception as e:
    print(f"Error: {e}")
