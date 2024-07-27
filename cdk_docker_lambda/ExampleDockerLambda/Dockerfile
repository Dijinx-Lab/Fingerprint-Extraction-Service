FROM amazon/aws-lambda-python:3.8

LABEL maintainer="Wesley Cheek"

# Install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the FastAPI application
COPY . ${LAMBDA_TASK_ROOT}/

# Set the command to start the FastAPI application with Mangum
CMD ["example_docker_lambda.handler"]