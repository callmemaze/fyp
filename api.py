from flask import Flask, jsonify, make_response, request, Response
from flask_restful import Resource, Api
from pymongo import MongoClient
from bson import json_util, ObjectId
import datetime
import bcrypt

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
users = db.get_collection('users')

api = Api(app)
utc_timestamp = datetime.datetime.utcnow()


class Alert(Resource):
    def post(self):
        name = request.form['name']
        date = request.form['date']
        image = request.form['image']
        status = request.form['status']
        collection.insert_one({'message': name,"date": date,"face": image, "expire": utc_timestamp, "status": status })
        return {"alert": "alert added"}
    
    def get(self):
        message = collection.find()
        response = json_util.dumps(message)
        return Response(response, mimetype="application/json")

class History(Resource):
    def post(self):
        name = request.form['name']
        status = request.form['status']
        date = request.form['date']
        history.insert_one({'message': name, "status": status,"date": date,"expire": utc_timestamp })
        return {"alert": "alert added"}
    
    def get(self):
        message = history.find()
        response = json_util.dumps(message)
        return Response(response, mimetype="application/json")

class DeleteHistory(Resource):
    def delete(self, id):
        history.delete_one({'_id': ObjectId(id)})
        return {"alert": "alert deleted"}

class Register(Resource):
    def post(self):
        data = request.get_json()
        print(data)
        email = data['email']
        password = data['password']
        name = data['name']
        user = users.find_one({'email': email})
        if user:
            return make_response(jsonify({"message": "User already exists"}), 401)
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(14))
        user_data = {'email': email, 'password': hashed, "name": name}
        response = json_util.dumps(user_data)
        users.insert_one(user_data)
        return Response(response, mimetype="application/json")

class Login(Resource):
    def post(self):
        data = request.get_json()
        email = data['email']
        password = data['password']
        user = users.find_one({'email': email})
        if not user:
            return make_response(jsonify({"message": "User does not exist"}), 401)
        if bcrypt.checkpw(password.encode('utf-8'), user['password']):
            response = json_util.dumps(user)
            return Response(response, mimetype="application/json")
        else:
            return {"message": "Incorrect password"}


api.add_resource(Alert, '/alert')
api.add_resource(History, '/history')
api.add_resource(DeleteHistory, '/history/<id>')
api.add_resource(Register, '/register')
api.add_resource(Login, '/login')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')

