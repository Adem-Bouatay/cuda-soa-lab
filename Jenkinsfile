pipeline {
    agent any
    
    environment {
        // IMPORTANT: Change this to your assigned student port
        STUDENT_PORT = "8001"
        IMAGE_NAME = "gpu-matrix-service"
        CONTAINER_NAME = "gpu-matrix-${STUDENT_PORT}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📦 Checking out source code...'
                git branch: 'master',
                    url: 'https://github.com/Adem-Bouatay/cuda-soa-lab.git'
            }
        }
        
        stage('GPU Sanity Test') {
            steps {
                echo '🔍 Installing required dependencies for cuda_test'
                sh '''
                    python3 -m pip install --user numpy numba-cuda[cu12]
                '''
                
                echo '🧪 Running CUDA sanity check...'
                sh '''
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
        
        stage('Test Docker Image') {
            steps {
                echo "🧪 Testing Docker image..."
                sh """
                    # Run a quick test container
                    docker run --rm --gpus all ${IMAGE_NAME}:latest python cuda_test.py
                """
            }
        }

        stage('Stop Old Container') {
            steps {
                echo "🛑 Stopping old container if exists..."
                sh """
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                """
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh """
                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        --gpus all \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        -p 800${STUDENT_PORT: -1}:8000 \
                        --restart unless-stopped \
                        ${IMAGE_NAME}:latest
                """
                
                echo "⏳ Waiting for service to start..."
                sh "sleep 10"
            }
        }
        
        stage('Health Check') {
            steps {
                echo "🏥 Performing health check..."
                sh """
                    curl -f http://localhost:${STUDENT_PORT}/health || exit 1
                    echo "✓ Service is healthy!"
                """
            }
        }
        
        stage('Verify Deployment') {
            steps {
                echo "✅ Verifying deployment..."
                sh """
                    echo "Container status:"
                    docker ps | grep ${CONTAINER_NAME}
                    
                    echo "\\nService endpoints:"
                    echo "  - Health: http://localhost:${STUDENT_PORT}/health"
                    echo "  - API: http://localhost:${STUDENT_PORT}/add"
                    echo "  - GPU Info: http://localhost:${STUDENT_PORT}/gpu-info"
                    echo "  - Metrics: http://localhost:${STUDENT_PORT}/metrics"
                    
                    echo "\\nTesting API endpoint:"
                    curl http://localhost:${STUDENT_PORT}/
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
                echo "Container logs:"
                docker logs ${CONTAINER_NAME} || true
            """
        }
        always {
            echo "🧾 Pipeline finished."
            sh """
                echo "Cleaning up old images..."
                docker image prune -f --filter "label=stage=builder" || true
            """
        }
    }
}
