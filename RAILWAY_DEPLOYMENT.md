# Railway Deployment Guide

## Prerequisites
1. Railway account (https://railway.app)
2. GitHub repository with your FastAPI backend code

## Deployment Steps

### 1. Connect Repository to Railway
1. Go to Railway dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository

### 2. Set Environment Variables
In Railway project settings, add these environment variables:
```
OPENAI_API_KEY=your_actual_openai_api_key
MODEL_NAME=gpt-4o-mini
UPLOAD_DIR=uploaded/
CACHE_DIR=cache/
PERSISTENCE_DIR=persistence/
PYTHONPATH=.
```

### 3. Configure Domain
1. Go to your Railway project
2. Click on "Settings" → "Domains"
3. Generate a Railway domain or add custom domain
4. Note the URL (e.g., `https://your-app-name.railway.app`)

### 4. Update Frontend Configuration
Update your Vercel frontend to point to the Railway backend URL:
```javascript
const API_BASE_URL = 'https://your-app-name.railway.app';
```

### 5. Test Deployment
Use the Railway URL in your Postman collection:
```
https://your-app-name.railway.app/health
https://your-app-name.railway.app/docs
```

## Troubleshooting

### Common Issues:
1. **404 Errors**: Check if the Railway app is properly deployed
2. **CORS Errors**: Verify allowed_origins in config.py includes your frontend URL
3. **Environment Variables**: Ensure OPENAI_API_KEY is set in Railway
4. **Port Issues**: Railway automatically sets PORT environment variable

### Logs:
Check Railway logs in the dashboard for debugging deployment issues.

## File Structure for Railway:
```
├── main.py (entry point)
├── railway.toml (Railway config)
├── requirements.txt (Python dependencies)
├── Dockerfile (container config)
├── app/ (FastAPI application)
└── .env.example (environment template)
```

## Production Checklist:
- [ ] Remove wildcard CORS (`"*"`) from allowed_origins
- [ ] Set proper environment variables
- [ ] Test all endpoints
- [ ] Monitor logs for errors
- [ ] Set up proper error handling