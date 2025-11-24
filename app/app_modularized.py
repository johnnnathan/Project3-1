from flask import Flask
from routes.main_routes import main_routes
from routes.process_routes import process_routes
from routes.visualize_routes import visualize_routes
from routes.processed_routes import processed_routes

def create_app():
    app = Flask(__name__)
    
    app.register_blueprint(main_routes)
    app.register_blueprint(process_routes)
    app.register_blueprint(visualize_routes)
    app.register_blueprint(processed_routes)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
