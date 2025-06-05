#!/usr/bin/env python3
import sys
import time
import random
from datetime import datetime
import signal
import os

# Add parent directory to path to import analytics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.analytics import Analytics

class AnalyticsDataGenerator:
    def __init__(self):
        self.analytics = Analytics()
        self.running = True
        self.browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']
        self.os_list = ['Windows 10', 'macOS', 'Ubuntu', 'iOS', 'Android']
        self.device_types = ['Desktop', 'Mobile', 'Tablet']
        self.languages = ['en-US', 'de-DE', 'fr-FR', 'es-ES', 'ja-JP']
        self.models = ['stable-diffusion', 'dall-e', 'midjourney']
        self.content_types = ['landscape', 'portrait', 'abstract', 'realistic']
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nShutting down generator...")
        self.running = False

    def generate_user_agent(self):
        browser = random.choice(self.browsers)
        os = random.choice(self.os_list)
        return f"{browser}/{random.randint(70, 120)}.0 ({os})"

    def generate_session_data(self):
        active_sessions = random.randint(5, 50)
        self.analytics.update_active_sessions(active_sessions)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Active sessions: {active_sessions}")

        # Generate new session data
        for _ in range(random.randint(1, 3)):
            user_agent = self.generate_user_agent()
            language = random.choice(self.languages)
            reference = f"REF{random.randint(1000, 9999)}"
            
            self.analytics.record_new_session(
                user_agent=user_agent,
                languages=language,
                reference=reference
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New session: {user_agent} | {language} | {reference}")

    def generate_image_data(self):
        # Generate image creation data
        for _ in range(random.randint(1, 5)):
            model = random.choice(self.models)
            content = random.choice(self.content_types)
            reference = f"IMG{random.randint(1000, 9999)}"
            
            # Start timer
            start_time = self.analytics.start_image_creation_timer()
            
            # Simulate image creation time
            time.sleep(random.uniform(5, 20))
            
            # Record image creation
            self.analytics.record_image_creation(
                count=1,
                model=model,
                content=content,
                reference=reference
            )
            self.analytics.record_prompt_usage(True, True)
            # Stop timer
            self.analytics.stop_image_creation_timer(start_time)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Image created: {model} | {content} | {reference}")

    def generate_user_tokens(self):
        # Update tokens for some random users
        for user_id in range(1, 6):
            tokens = random.randint(0, 100)
            self.analytics.update_user_tokens(f"user_{user_id}", tokens)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] User tokens: user_{user_id} | {tokens}")

    def run(self):
        print("Starting analytics data generator...")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                # Generate various metrics
                self.generate_session_data()
                self.generate_image_data()
                self.generate_user_tokens()
                
                # Wait for next iteration
                interval = random.randint(30, 60)
                print(f"\nWaiting {interval} seconds for next update...\n")
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error generating data: {e}")
                time.sleep(5)

if __name__ == "__main__":
    #analytics = Analytics()
    #analytics.record_prompt_usage(True, True)
    #input("ha")
    generator = AnalyticsDataGenerator()
    generator.run()
