pipeline {
    agent any


    stages {

        stage('GPU Sanity Test') {
            steps {
                echo 'Installing required dependencies for cuda_test'
                // TODO: write here
                echo 'Running CUDA sanity check...'
                // TODO: write here
            }
        }


        stage('Build Docker Image') {
            steps {
                // TODO: write here
                echo "🐳 Building Docker image with GPU support..."
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                // TODO: write here
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
