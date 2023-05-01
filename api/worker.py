import os
import pika
import http.server
import socketserver
import threading
import json
from commands import commands

connection = pika.BlockingConnection(
    pika.URLParameters(
        os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672")
    )
)

channel = connection.channel()

channel.queue_declare(queue="features_rpc_queue")


def on_request(ch, method, props, body):
    try:
        print(body)
        command, args = json.loads(body)
        response = json.dumps(commands[command](*args))
    except Exception as e:
        print(e)
        response = json.dumps(
            {"exception_type": type(e).__name__, "exception_args": e.args}
        )

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=response,
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="features_rpc_queue", on_message_callback=on_request)


class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if connection and connection.is_open and channel and channel.is_open:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(503)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"RabbitMQ connection is closed or channel is not open")


httpd = socketserver.TCPServer(("", 8000), HealthCheckHandler)

http_thread = threading.Thread(target=httpd.serve_forever)
http_thread.daemon = True
http_thread.start()

try:
    print("Awaiting RPC requests")
    channel.start_consuming()
finally:
    httpd.shutdown()
    httpd.server_close()
