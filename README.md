# Running the Project with Docker
## Setup
To simplify the setup and execution of the SignEval project, Docker and Docker Compose can be utilized. Follow the steps below to build and run the project using Docker:
1. **Clone the Repository**
   ```bash
   git clone git clone https://github.com/yourusername/Distributed-ASL.git
   cd Distributed-ASL
   ```
2. **Create Virtual Environments and install necessary requirements**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
   
## Prerequisites

- Ensure Docker and Docker Compose are installed on your system.
- Verify that the required ports (5000) are available.

## Steps to Build and Run

1. **Build the Docker Images**

   Navigate to the project directory and execute the following command to build the Docker images:

   ```bash
   docker-compose build
   ```

2. **Run the Services**

   Start the services defined in the `docker-compose.yml` file:

   ```bash
   docker-compose up -d recognition1 recognition2 recognition3 load_balancer dashboard database
   ```

   This will:
   - Launch the `signeval` service, exposing it on port `5000`.
   - Start a PostgreSQL database service for backend support.

3. **Define the Port of the Load Balancer**
   On the Command Line type:
   ```bash
   $env:LOAD_BALANCER_URL = "http://localhost:5001"  
   ```
4. **Enable Debugging Mode**
   On the Command Line type:
   ```bash
   $env:LOAD_BALANCER_URL = "http://localhost:5001"
   ```
5. **Access the Application**
   Run the Python frontend file , to run the frontend locally:
   ```bash
   python frontend/app.py   
   ```
   Open your web browser and navigate to `http://localhost:5000` to access the SignEval application.

6. **Access the Dashboard**
   Open your web browser and navigate to `http://localhost:5002` to access the Dashboard
   Choose the load balancing algorithm from the dropdown and observe the statistics.

7. **Run the Monolithic Application**
   Find the `src/app.py` and run the application using:
   ```bash
   python signeval/app.py
   ```
   Use the Monolithic Application to compare and research into the perks of Distributed Working Systems.
## Configuration

- The `signeval` service depends on the `database` service, which is configured with the following environment variables:
  - `POSTGRES_USER`: `user`
  - `POSTGRES_PASSWORD`: `password`

- Modify these variables in the `docker-compose.yml` file if needed.

## Notes

- The application code is located in the `/app` directory within the container.
- Logs and database data are stored in the respective volumes defined in the `docker-compose.yml` file.

By using Docker, you can ensure a consistent and isolated environment for running the SignEval project.
