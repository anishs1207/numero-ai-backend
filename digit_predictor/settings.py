import os
import environ

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = environ.Env(
    DEBUG=(bool, False),
    SECRET_KEY=(str, None),
)

env_file_path_check = os.path.join(BASE_DIR, '.env')


environ.Env.read_env(env_file_path_check)

DEBUG = env('DEBUG')
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY') # Loaded from environment variable

# You can remove your original `print(DEBUG, SECRET_KEY)` line now,
# as the debugging block above covers it.
# print(DEBUG, SECRET_KEY) # <--- REMOVE OR COMMENT OUT THIS LINE


# SECURITY WARNING: don't run with debug turned on in production!
# DEBUG = env('DEBUG') # This line is redundant, remove it. DEBUG is set above.

# Define allowed hosts for your application (your backend's domain/IP)
# REPLACE PLACEHOLDERS with your actual production domain/IP
ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'numero-ai.vercel.app', 'your-backend-domain.com', 'your_server_ip']

INSTALLED_APPS = [
    # Remove these if your project truly doesn't use a database, admin, auth, sessions, or messages
    # For now, keeping them as they are common and might be implicitly used or desired later.
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles', # Essential for static files
    'rest_framework', # Your DRF app
    'digit_app',      # Your custom Django app
    'corsheaders',    # For CORS handling
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware', # CRITICAL: Moved up to the correct position
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# --- CORS Configuration ---
# CRITICAL: Do NOT use CORS_ALLOW_ALL_ORIGINS = True in production.
# List your specific frontend origins.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",        # For local frontend development
    "https://numero-ai.vercel.app", # Your deployed Vercel frontend
    # Add other specific frontend origins if your frontend is deployed elsewhere
]

# If your frontend needs to send credentials (cookies, auth headers like Authorization),
# uncomment the line below. If so, CORS_ALLOWED_ORIGINS cannot contain '*'.
# CORS_ALLOW_CREDENTIALS = True


ROOT_URLCONF = 'digit_predictor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'digit_predictor.wsgi.application'


# --- Database Configuration ---
# CRITICAL: DO NOT USE SQLITE IN PRODUCTION.
# If you need a database (e.g., for Django Admin/Auth), configure PostgreSQL/MySQL.
# Using django-environ for DATABASE_URL parsing.
DATABASES = {
    'default': env.db('DATABASE_URL', default='sqlite:///db.sqlite3')
}
# Recommended for production DB connections
DATABASES['default']['CONN_MAX_AGE'] = 600 # seconds


# --- Password Validation (Relevant if django.contrib.auth is used) ---
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# --- Internationalization ---
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True # Deprecated in Django 4.0+, USE_TZ handles this
USE_TZ = True


# --- Static Files (CSS, JavaScript, Images) ---
STATIC_URL = '/static/'
# CRITICAL: Define STATIC_ROOT for collecting static files in production
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles_collected')

# If your app handles user-uploaded media files (e.g., if users upload images to be recognized):
# MEDIA_URL = '/media/'
# MEDIA_ROOT = os.path.join(BASE_DIR, 'mediafiles_uploaded') # Store user-uploaded files here