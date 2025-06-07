# Gmail Setup Guide for SereniTeach Crisis Alerts

## Current Issue
The email credentials provided are not working with Gmail's SMTP server. Gmail requires specific security configurations for external applications.

## Required Steps:

### 1. Enable 2-Step Verification
- Go to your Google Account settings
- Navigate to Security
- Enable 2-Step Verification if not already enabled

### 2. Generate App Password
- In Google Account settings > Security
- Find "2-Step Verification" section
- Click on "App passwords"
- Select "Mail" as the app type
- Generate a 16-character app password
- Use this password (not your regular Gmail password)

### 3. Required Credentials Format:
```
MAIL_USERNAME: your-full-email@gmail.com
MAIL_PASSWORD: xxxx xxxx xxxx xxxx (16-character app password)
```

### 4. Alternative Solutions:
If you prefer not to use Gmail, we can:
- Use a different email service (SendGrid, Mailgun)
- Set up SMTP through your school's email system
- Use webhook notifications to another service

## Current Status:
- Crisis detection: ✅ Working
- Email configuration: ❌ Needs proper Gmail app password
- Voice interface: ✅ Working
- Alert system integration: ✅ Working