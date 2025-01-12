import requests
import json
import os

def __get_webwook_url__():
    # webhook url の読み込み
    # github で webhook の url を見れなくするために直接打ち込まず、
    # git の追跡をしない __pycache__/ の中に url の書かれた txt を入れてそれを読み取る
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, "__pycache__/webhook_url.txt")
    with open(file_path, encoding="UTF-8") as f:
        webhook_url = f.read()
    return webhook_url

# Discord や Slack に通知をする関数
def text(text ="計算が終わりました"):

    webhook_url = __get_webwook_url__()

    #内容
    payload = {"content": text}
    headers      = {"Content-Type": "application/json"}

    #リクエスト送信
    requests.post(webhook_url, json.dumps(payload), headers=headers)


# https://blog.shikoan.com/requests_discord/ をコピペしただけ
# ちゃんと Discord api と json 理解して自分で書くべきではある
def image(text, image_path):

    webhook_url = __get_webwook_url__()

    payload = {
        "content": text
    }

    # 画像ファイルを Discord に送るためのおまじない
    file_list = [image_path]
    multiple_files = []
    for i, image_file in enumerate(file_list):
        multiple_files.append((
            f"files[{i}]", (f"image{i+1}.jpg", open(image_file, "rb"), "image/jpg")
        ))

    # リクエスト送信
    response = requests.post(webhook_url, data={"payload_json": json.dumps(payload)}, files=multiple_files)
    print("sent message. HTTP Response:{}\n".format(response.status_code))

    # 開いたファイルを閉じる
    for name, filetuple in multiple_files:
        if isinstance(filetuple, tuple) and filetuple[1]:
            filetuple[1].close()
