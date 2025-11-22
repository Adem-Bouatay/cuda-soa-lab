pipeline {
    agent any

    environment {
        IMAGE_NAME = 'gpu-service'
        CONTAINER_NAME = 'gpu-matrix-service'
        STUDENT_PORT = '8020'  // Change this to your assigned port
    }

    stages {

        stage('GPU Sanity Test') {
            steps {
                echo 'Creating virtual environment and installing dependencies'
                sh '''
                    # Create virtual environment if it doesn't exist
                    python3 -m venv venv

                    # Activate and install dependencies
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install numba-cuda[cu12] numpy
                '''
                echo 'Running CUDA sanity check...'
                sh '''
                    # Run test within virtual environment
                    . venv/bin/activate
                    python3 cuda_test.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh '''
                    docker build -t ${IMAGE_NAME}:latest .
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh '''
                    # Stop and remove existing container if running
                    docker stop ${CONTAINER_NAME} 2>/dev/null || true
                    docker rm ${CONTAINER_NAME} 2>/dev/null || true

                    # Run new container with GPU support
                    docker run --gpus all \
                        -d \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        -p 8000:8000 \
                        --name ${CONTAINER_NAME} \
                        --restart unless-stopped \
                        ${IMAGE_NAME}:latest

                    # Wait for container to be healthy
                    sleep 5

                    # Check container status
                    docker ps | grep ${CONTAINER_NAME}
                '''
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
        }
        always {
            echo "🧾 Pipeline finished."
        }
    }
}
