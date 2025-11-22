pipeline {
    agent any
    
    environment {
        STUDENT_PORT = "8699"
        IMAGE_NAME = "gpu-matrix-service"
        CONTAINER_NAME = "gpu-matrix-${STUDENT_PORT}"
    }

    stages {

        stage('GPU Sanity Test') {
            steps {
                echo 'Creating virtual environment'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    python3 -m pip install --upgrade pip
                '''
                echo 'Installing required dependencies for cuda_test'
                sh '''
                    . venv/bin/activate
                    pip install numpy numba-cuda[cu12]
                '''
                echo 'Running CUDA sanity check...'
                sh '''
                    . venv/bin/activate
                    python3 cuda_test.py
                '''
            }
        }


        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh """
                    docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .
                    docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest
                """
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh """
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        --gpus all \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        -p 8000:8000 \
                        --restart unless-stopped \
                        ${IMAGE_NAME}:latest
                    sleep 10
                    curl -f http://localhost:${STUDENT_PORT}/health || exit 1
                """
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
            echo "Service is running on port ${STUDENT_PORT}"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
            sh """
                docker logs ${CONTAINER_NAME} || true
            """
        }
        always {
            echo "🧾 Pipeline finished."
            sh """
                docker image prune -f || true
            """
        }
    }
}