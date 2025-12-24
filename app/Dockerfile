# משתמשים בגרסה רזה של פייתון כדי שהאימג' יהיה קל
FROM python:3.10-slim

# הגדרת תיקיית עבודה
WORKDIR /app

# התקנת תלויות מערכת (אם צריך)
RUN apt-get update && apt-get install -y build-essential

# העתקת קובץ הדרישות והתקנת ספריות פייתון
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת שאר הקוד
COPY . .

# חשיפת פורטים (רק לתיעוד, ה-Compose קובע בפועל)
EXPOSE 8000 8501