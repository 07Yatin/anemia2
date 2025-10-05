# Anemia Detection WebApp - Render Deployment Guide

## Prerequisites
- GitHub Account
- Render Account
- Python 3.10+

## Deployment Steps

### 1. GitHub Repository
- Ensure your code is pushed to a GitHub repository
- Verify `requirements.txt` is up to date
- Confirm `Procfile` and `render.yaml` exist

### 2. Render Setup
1. Log in to [Render](https://render.com/)
2. Click "New +" and select "Web Service"
3. Choose "Deploy an existing repository from GitHub"

### 3. Configuration Details
- **Name**: anemia-detection-webapp
- **Environment**: Python
- **Branch**: main
- **Runtime**: Python 3.10
- **Deployment Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn gradioApp:app --host 0.0.0.0 --port $PORT`

### 4. Environment Variables
- `PORT`: Automatically managed by Render
- Add any additional environment variables required by your app

### 5. Deployment Considerations
- Free tier spins down after 15 minutes of inactivity
- First load might take 30-60 seconds to restart
- Recommended for testing and light usage

### Troubleshooting
- Check Render deployment logs
- Verify all dependencies in `requirements.txt`
- Ensure start command is correct
- Check Python version compatibility

## Recommended Upgrades
- Consider Render Pro for continuous deployment
- Add monitoring and performance tracking
