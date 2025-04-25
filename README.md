# Running the Project with Docker

To simplify the setup and execution of the SignEval project, Docker and Docker Compose can be utilized. Follow the steps below to build and run the project using Docker:

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
   docker-compose up
   ```

   This will:
   - Launch the `signeval` service, exposing it on port `5000`.
   - Start a PostgreSQL database service for backend support.

3. **Access the Application**

   Open your web browser and navigate to `http://localhost:5000` to access the SignEval application.

## Configuration

- The `signeval` service depends on the `database` service, which is configured with the following environment variables:
  - `POSTGRES_USER`: `user`
  - `POSTGRES_PASSWORD`: `password`

- Modify these variables in the `docker-compose.yml` file if needed.

## Notes

- The application code is located in the `/app` directory within the container.
- Logs and database data are stored in the respective volumes defined in the `docker-compose.yml` file.

By using Docker, you can ensure a consistent and isolated environment for running the SignEval project.