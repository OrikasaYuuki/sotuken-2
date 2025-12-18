import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import json
import time
import io
import os


os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
cv2.imshow = lambda *args, **kwargs: None
cv2.namedWindow = lambda *args, **kwargs: None

# ====== 設定 ======
# Gemini APIキー（安全のためファイルから読み込むのが望ましい）
MODEL_PATH = "./last.pt"
MEMORY_PATH = "./memory.txt"
JSON_PATH = "./json/result.json"

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

with open("memory.txt", "r" , encoding="utf-8") as f:
    memory = f.read()
    
# YOLOモデル読み込み
model = YOLO(MODEL_PATH)


# ====== sidebar UI ======
st.sidebar.title("test")
st.sidebar.write("画像をアップロードすると、物体検出と被害予測を行います。")
uploaded_file = st.sidebar.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

# ====== main UI ======
st.title("災害被害予測システム")
st.write("画像をアップロードすると、物体検出と被害予測を行います。")

with st.expander('変更記録'):
    st.text(memory)



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="入力画像", use_column_width=True)

    # ===== YOLO検出 =====
    st.subheader("物体検出中...")
    results = model(image)

    if len(results[0].boxes) == 0:
        st.warning("画像から物体が検出されませんでした。" \
        "申し訳ありませんが、別の画像でお試しください。")
        st.stop()   # ← ここで処理を終了（Geminiへ進まない）
    
    # 検出結果の可視化
    res_img = results[0].plot()
    st.image(res_img, caption="検出結果", use_column_width=True)
    # 検出結果をJSONに変換
    detections = []
    class_names = [
        "bed", "sofa", "chair", "table", "lamp",
        "tv", "laptop", "wardrobe", "window",
        "door", "potted plant", "photo frame"
    ]

    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        detections.append({
            "class_id": cls,
            "class_name": class_names[cls]
        })

    json_output = [{
        "image_id": uploaded_file.name,
        "detections": detections
    }]

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    # st.write("検出された物体:", [d["class_name"] for d in detections])

    # ===== Geminiで被害予測 =====
    st.subheader("Geminiによる被害予測結果")

    prompt = f"""
    以下のJSONファイルについて分析し、重複したデータは1つにまとめてください。
    その上でclass_nameを日本語に直し、災害時に想定される被害を説明してください。
    JSONの中身は表示せず、被害説明のみを日本語で出力してください。
    各物体の被害説明以外の文章は出力しないでください。
    各被害説明は過剰書きで出力してください。
    ---
    {json.dumps(json_output, ensure_ascii=False, indent=2)}
    ---
    """

    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
    with st.spinner("Geminiが被害を推定中です..."):
        response = model_gemini.generate_content(prompt)
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
    st.success("被害予測が完了しました")
    st.write(response.text)

















