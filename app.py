import cv2
import base64
import numpy as np
import requests
import time
import json
import hashlib

from crnn import CRNN
from dbnet import DBNet
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'this is a secret'
CORS(app, supports_credentials=True)

def order_points_clockwise(box_points):
    points = np.array(box_points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    quad_box = np.zeros((4, 2), dtype=np.float32)
    quad_box[0] = points[np.argmin(s)]
    quad_box[2] = points[np.argmax(s)]
    quad_box[1] = points[np.argmin(diff)]
    quad_box[3] = points[np.argmax(diff)]
    return quad_box

def get_patch(page, points):
    points = order_points_clockwise(points)
    page_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]))
    )
    page_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32([
        [0, 0], [page_crop_width, 0], 
        [page_crop_width, page_crop_height],[0, page_crop_height]
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    return cv2.warpPerspective(
        page, M, (page_crop_width, page_crop_height), 
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )

def hash_bytes(bytes_data):
    hash_object = hashlib.sha256(bytes_data)
    hash_str = hash_object.hexdigest()
    return hash_str

def load_models():
    det_model = DBNet()
    rec_model = CRNN()
    det_model.model.load_weights('./assets/DBNet.h5')
    rec_model.model.load_weights('./assets/CRNN.h5')
    return det_model, rec_model

det_model, rec_model = load_models()

def readb64(uri):
    encoded_data = uri
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def the_translate(text):
    url = 'https://api.clc.hcmus.edu.vn/nom_translation/90/1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Content-Type': 'application/json'
    }
    payload = {
        'nom_text': text
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        time.sleep(0.1)
        
        if response.status_code == 200:
            try:
                result = json.loads(response.text)['sentences']
                result = result[0][0]['pair']['modern_text']
                return result
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f'[ERR] JSON parsing error for text "{text}": {e}')
                return 'Không thể dịch văn bản này.'
        else:
            print(f'[ERR] Yêu cầu thất bại với mã trạng thái {response.status_code} cho văn bản "{text}": {response.text}')
            return 'Không thể dịch văn bản này.'
    except requests.RequestException as e:
        print(f'[ERR] Lỗi yêu cầu cho văn bản "{text}": {e}')
        return 'Không thể dịch văn bản này.'


@app.route("/", methods=["GET"])
def main():
    return jsonify({
        "message": "Hello hihi"
    })

@app.route("/inference", methods=["POST"])
def inference():
    raw_image = request.json['img_base64']
    img_processed = readb64(raw_image)
    img_bytes = img_processed.tobytes()
    boxes = det_model.predict_one_page(img_processed)

    translation = {}
    
    for idx, box in enumerate(boxes):
        patch = get_patch(img_processed, box)
        nom_text = rec_model.predict_one_patch(patch).strip()
        modern_text = the_translate(nom_text).strip()

        points = sum(box.tolist(), [])
        points = [str(round(p)) for p in points]
        final_xy = []
        for i in range(0, len(points), 2):
            final_xy.append({
                "x": points[i],
                "y": points[i+1]
            })

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        raw_ret_image = base64.b64encode(buffer)
        translation[idx] = {
            "nom_text": nom_text,
            "modern_text": modern_text,
            "points": final_xy,
            "patch_img_base64": raw_ret_image.decode("utf-8")
        }

    return jsonify({
        "translation": translation,
    })

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')    