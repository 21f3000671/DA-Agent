def handler(request):
    """
    Ultra minimal handler for Vercel debugging
    """
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": '{"message": "Ultra minimal test working", "status": "ok"}'
    }