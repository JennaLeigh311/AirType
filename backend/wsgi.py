"""
WSGI entry point for AirType backend
"""

import os
from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    socketio.run(app, debug=True, host='0.0.0.0', port=port)
