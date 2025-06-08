import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("Starting application...")

# Set up database
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "sereni-teach-voice-assistant")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the PostgreSQL database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Configure Flask-Mail for email alerts
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'noreply@sereniteach.app')

# Initialize extensions
db.init_app(app)
mail = Mail(app)

# Import routes after the app is created to avoid circular imports
with app.app_context():
    from routes import *  # noqa: F401, F403
    import models  # noqa: F401

    # Create database tables
    db.create_all()
    logger.info("Database tables created")
    
    # Import protocol data if not already imported
    try:
        from import_protocols import run_imports
        import os
        
        if os.path.exists('attached_assets/sama_protocols.csv'):
            # Check if we already have protocols imported
            from models import Protocol
            if Protocol.query.count() < 5:  # Only import if we have fewer than 5 protocols
                logger.info("Importing protocol data...")
                run_imports()
                logger.info("Protocol data import completed")
            else:
                logger.info("Protocol data already imported, skipping")
    except Exception as e:
        logger.error(f"Error importing protocols: {str(e)}")
        # Continue even if import fails to ensure application starts
