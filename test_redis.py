try:
    import redis
    print("Redis is already installed")
except ImportError:
    print("Redis is NOT installed")
