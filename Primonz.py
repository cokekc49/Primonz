from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import requests
import predict
'''import nest_asyncio
nest_asyncio.apply()
__import__('IPython').embed()'''

app = FastAPI()

'''@app.get("/index")
def my_function(name:str):
  return f"Hello {name} !"'''

@app.post("/api/predict")
def imgurl(p_image_url: str):
  #file#Image.open(image.file)
  r = requests.get(p_image_url, allow_redirects=True)
  with open("file2.png",'wb') as f:
      f.write(r.content)
  name = "file2.png"
  char = predict.answer(name)
  return {"name":str(char)}

'''def predict_image(file:UploadFile = File(...)):
  #print('aaa')
  name = file.filename #.split('.')[-1] in ("jpg", "jpeg", "png","jfif")
  char = predict.answer(name)
  return {"name":str(char)}'''

if __name__ == "__main__":
  uvicorn.run(app, host="localhost", port=2034, debug=True)