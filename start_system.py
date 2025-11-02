#!/usr/bin/env python3
"""
Startup Script for Cyber Threat Detection System
Starts both backend API server and frontend React app
"""

import os
import sys
import subprocess
import time
import threading
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemStarter:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def start_backend(self):
        """Start the Flask backend API server"""
        try:
            logger.info("🚀 Starting Backend API Server...")
            
            # Change to backend directory
            backend_dir = Path("backend")
            if not backend_dir.exists():
                logger.error("❌ Backend directory not found!")
                return False
            
            # Start the Flask server
            self.backend_process = subprocess.Popen(
                [sys.executable, "app_simple.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                logger.info("✅ Backend API server started successfully on http://localhost:5000")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                logger.error(f"❌ Backend failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting backend: {str(e)}")
            return False
    
    def start_frontend(self):
        """Start the React frontend"""
        try:
            logger.info("🚀 Starting React Frontend...")
            
            # Change to frontend directory
            frontend_dir = Path("frontend")
            if not frontend_dir.exists():
                logger.error("❌ Frontend directory not found!")
                return False
            
            # Check if node_modules exists
            if not (frontend_dir / "node_modules").exists():
                logger.info("📦 Installing frontend dependencies...")
                install_process = subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    capture_output=True,
                    text=True
                )
                if install_process.returncode != 0:
                    logger.error(f"❌ Failed to install dependencies: {install_process.stderr}")
                    return False
                logger.info("✅ Frontend dependencies installed")
            
            # Start the React development server
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for server to start
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                logger.info("✅ React frontend started successfully on http://localhost:3000")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                logger.error(f"❌ Frontend failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting frontend: {str(e)}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            time.sleep(2)
            
            # Check backend
            if self.backend_process and self.backend_process.poll() is not None:
                logger.warning("⚠️ Backend process stopped unexpectedly")
                self.running = False
                break
            
            # Check frontend
            if self.frontend_process and self.frontend_process.poll() is not None:
                logger.warning("⚠️ Frontend process stopped unexpectedly")
                self.running = False
                break
    
    def stop_all(self):
        """Stop all running processes"""
        logger.info("🛑 Stopping all processes...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        logger.info("✅ All processes stopped")
    
    def run(self):
        """Run the complete system"""
        logger.info("=" * 60)
        logger.info("🔒 CYBER THREAT DETECTION SYSTEM")
        logger.info("=" * 60)
        
        # Check if required files exist
        required_files = [
            "backend/data/consolidated/All_Network_Complete.csv",
            "backend/models/aca_rf_model.pkl",
            "backend/models/fuzzy_rf_model.joblib",
            "backend/models/intrudtree_model.joblib"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning("⚠️ Some required files are missing:")
            for file in missing_files:
                logger.warning(f"   - {file}")
            logger.warning("Please ensure all models are trained before starting the system")
        
        # Start backend
        if not self.start_backend():
            logger.error("❌ Failed to start backend. Exiting.")
            return False
        
        # Start frontend
        if not self.start_frontend():
            logger.error("❌ Failed to start frontend. Stopping backend.")
            self.stop_all()
            return False
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
            self.stop_all()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("=" * 60)
        logger.info("🎉 SYSTEM STARTED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📱 Frontend: http://localhost:3000")
        logger.info("🔧 Backend API: http://localhost:5000")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the system")
        logger.info("=" * 60)
        
        # Monitor processes
        self.monitor_processes()
        
        return True

def main():
    """Main function"""
    starter = SystemStarter()
    
    try:
        success = starter.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt received")
        starter.stop_all()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        starter.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
