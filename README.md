# Information Retrieval Project
## Introduction
In this project, information retrieval methods are used to find a query in a collection of ferdosi's poems.
## Prerequisites
In order to run this project, you only need docker and docker compose.
## How to run this project
To run this project, you need to run the following commands:
```bash
docker compose -p ir up --build -d
```
By running this command, every requirement is installed and the project is ready to run.
To see the webpage, browse to http://localhost:8080/

After you're done with the project, you can run the following command to stop the project:
```bash
docker compose -p ir down --rmi all
```