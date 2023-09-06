from  pymongo import MongoClient
from  datetime import datetime

datetime_now = datetime.now() # pass this to a MongoDB doc
print ("datetime_now:", datetime_now)

myclient = MongoClient("mongodb+srv://Himanshi:BvJQzuRUSBKEzINq@cluster0.kfs5kzr.mongodb.net/")
mybd = myclient["wait_data_base"]
mycol = mybd['wait_collection']

mylist = [ {'_id': 20, 'license_plate_number': 4565, 'speed(Km/h)': 54},
{'_id': 21, 'license_plate_number': 4155, 'speed(Km/h)': 51,"datetime":datetime_now},
{'_id': 22, 'license_plate_number': 1455, 'speed(Km/h)': 60,"datetime":datetime_now},
{'_id': 23, 'license_plate_number': 8152, 'speed(Km/h)': 65,"datetime":datetime_now},
{'_id': 24, 'license_plate_number': 8655, 'speed(Km/h)': 56,"datetime":datetime_now},
{'_id': 25, 'license_plate_number': 5544, 'speed(Km/h)': 55,"datetime":datetime_now},
{'_id': 26, 'license_plate_number': 4632, 'speed(Km/h)': 86,"datetime":datetime_now},
{'_id': 27, 'license_plate_number': 4532, 'speed(Km/h)': 76,"datetime":datetime_now},
{'_id': 28, 'license_plate_number': 7685, 'speed(Km/h)': 89,"datetime":datetime_now},
{'_id': 29, 'license_plate_number': 7865, 'speed(Km/h)': 65,"datetime":datetime_now},
{'_id': 30, 'license_plate_number': 1256, 'speed(Km/h)': 78,"datetime":datetime_now}]
mycol.insert_many(mylist)
 
for x in mycol.find():
    print(x)
#BvJQzuRUSBKEzINq