import pyrebase


config = {
    "apiKey": "AIzaSyDTqeM5PZd2mzBsYxzwQArkcFgRfUt4RlI",
  "authDomain": "upload-images-aae3e.firebaseapp.com",
  "projectId": "upload-images-aae3e",
  "storageBucket": "upload-images-aae3e.appspot.com",
  "messagingSenderId": "189694713606",
  "appId": "1:189694713606:web:b993e8b9eb36f99b8e0887",
  "measurementId": "G-GF5B84G1PN",
  "serviceAccount": "serviceAccount.json",
  "databaseURL": "https://console.firebase.google.com/project/upload-images-aae3e/database/upload-images-aae3e-default-rtdb/data/~2F"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
storage.child("img2.jpg").put("F_9.jpg")
print("upload the image successfully")





# import firebase_admin
# from firebase_admin import credentials

# cred = credentials.Certificate("/home/himanshi/License-Plate-Recognition/serviceAccount.json")
# firebase_admin.initialize_app(cred,{'databaseURL': 'https://console.firebase.google.com/project/upload-images-aae3e/database/upload-images-aae3e-default-rtdb/data/~2F'})