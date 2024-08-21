# Gunakan image Python sebagai basis
FROM python:3.9-slim

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Buat directory di container untuk aplikasi
WORKDIR /app

# Copy semua file ke dalam directory di container
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port 8080 untuk aplikasi
EXPOSE 8080

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "-b", ":8080", "app:app"]
