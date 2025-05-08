#Use the official python base image
FROM python:3.9


#Set the working directory inside the container
WORKDIR /app 

# Copy the current directory contents into the container at /app
COPY . /app

# COPY requiremenets.txt .

#install requirements.txt 
RUN pip install -r requirements.txt 

#Copy the application code to the working directory 
COPY . . 

#
ENV NAME venv

#Expose the port on which the application form 
EXPOSE 8000


##Run the main.py when the container launches 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]