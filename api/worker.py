import os
import pika
import http.server
import socketserver
import threading
import json
import traceback
from commands import commands
from common import JOB_QUEUE_NAME


class RpcServer:
    def __init__(self, queue_name=JOB_QUEUE_NAME):
        self.connection = pika.BlockingConnection(
            pika.URLParameters(
                os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672")
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue_name)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=queue_name, on_message_callback=self.on_request
        )

    def on_request(self, ch, method, props, body):
        try:
            body_str = str(body)
            if len(body_str) > 60:
                body_str = body_str[:50] + "..." + body_str[-10:]
            print(body_str)
            command, args = json.loads(body)
            response = json.dumps(commands[command](*args))
        except Exception as e:
            traceback.print_exc()
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

    def start(self):
        print("Awaiting RPC requests")
        self.channel.start_consuming()
        print("Stopping")


class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.rpc_server = kwargs.pop("rpc_server", None)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if (
            self.rpc_server.connection
            and self.rpc_server.connection.is_open
            and self.rpc_server.channel
            and self.rpc_server.channel.is_open
        ):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(503)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"RabbitMQ connection is closed or channel is not open")


if __name__ == "__main__":
    rpc_server = RpcServer()

    httpd = socketserver.TCPServer(
        ("", 8000),
        lambda *args, **kwargs: HealthCheckHandler(
            *args, **kwargs, rpc_server=rpc_server
        ),
    )

    http_thread = threading.Thread(target=httpd.serve_forever)
    http_thread.daemon = True
    http_thread.start()
    try:
        rpc_server.start()
    finally:
        httpd.shutdown()
        httpd.server_close()
