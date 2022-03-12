from distutils.command.clean import clean
from time import time
from unicodedata import name
from flask import Flask, jsonify, request, Response
from flask_restful import Resource, Api
from pymongo import MongoClient
import json
from bson import json_util
import datetime
app = Flask(__name__)
CONNECTION_STRING = "mongodb+srv://Maze:Maze@cluster0.bjjtz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STRING)
db = client.get_database('myFirstDatabase')
collection = db.get_collection('alerts')
collection.create_index("expire", expireAfterSeconds=3*60)
history = db.get_collection('history')
history.create_index("expire", expireAfterSeconds=2592000)
unlock = db.get_collection('unlock')
unlock.create_index("expire", expireAfterSeconds=3*60)
api = Api(app)
utc_timestamp = datetime.datetime.utcnow()


class Alert(Resource):
    def post(self):
        name = request.form['name']
        collection.insert_one({'message': name, "expire": utc_timestamp })
        return {"alert": "alert added"}
    
    def get(self):
        message = collection.find()
        response = json_util.dumps(message)
        return Response(response, mimetype="application/json")

class History(Resource):
    def post(self):
        name = request.form['name']
        status = request.form['status']
        history.insert_one({'message': name, "status": status,  "expire": utc_timestamp })
        return {"alert": "alert added"}
    
    def get(self):
        message = history.find()
        response = json_util.dumps(message)
        return Response(response, mimetype="application/json")

class Unlock(Resource):
    def post(self):
        name = request.get_json()
        print(request.data)
        print(name)
        unlock.insert_one({'message': name,  "expire": utc_timestamp })
        return {"alert": "alert added"}
    
    def get(self):
        message = unlock.find()
        response = json_util.dumps(message)
        return Response(response, mimetype="application/json")


api.add_resource(Alert, '/alert')
api.add_resource(History, '/history')
api.add_resource(Unlock, '/unlock')

if __name__ == '__main__':
    app.run(debug=True)

