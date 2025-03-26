from fastapi import FastAPI, Request # type: ignore
import uvicorn # type: ignore
from sllurp.llrp import LLRPReaderClient
import asyncio
from contextlib import asynccontextmanager
import logging
import threading
import queue

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Connect to the R420 Reader
READER_IP = "169.254.1.1"
PORT = 14150

# Global client and queue
client = None
tag_queue = queue.Queue()

# Callback to handle tag reports
def tag_callback(reader, tags):
    """Queue tag data from sllurp thread."""
    logger.debug(f"Tag report received in thread {threading.current_thread().name}: {tags}")
    if tags:
        tag_queue.put(tags)

async def process_tags():
    """Process queued tags in the main asyncio loop."""
    logger.debug("Starting process_tags task")
    while True:
        if not tag_queue.empty():
            tags = tag_queue.get()
            logger.debug(f"Dequeued tags: {tags}")
            for tag in tags:
                epc = tag.get('EPC', 'Unknown')
                antenna = tag.get('AntennaID', 'Unknown')
                rssi = tag.get('PeakRSSI', 'Unknown')
                phase_angle = tag.get('PhaseAngle', 'Unknown')
                frequency = tag.get('ChannelIndex', 'Unknown')
                read_count = tag.get('TagSeenCount', 0)
                doppler_frequency = tag.get('DopplerFrequency', 'Unknown')
                timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')  # Using LastSeenTimestampUTC
                logger.info(
                    f"Timestamp: {timestamp}\n"
                    f"EPC: {epc}\n"
                    f"Antenna: {antenna}\n"
                    f"RSSI: {rssi}\n"
                    f"PhaseAngle: {phase_angle}\n"
                    f"Frequency: {frequency}\n"
                    f"ReadCount: {read_count}\n"
                    f"DopplerFrequency: {doppler_frequency}\n"
                )
        await asyncio.sleep(0.01)  # Short sleep to yield control

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    logger.info("🔵 Connecting to R420 Reader at %s:%d", READER_IP, PORT)
    client = LLRPReaderClient(host=READER_IP, port=PORT)
    client.add_tag_report_callback(tag_callback)
    try:
        await asyncio.get_event_loop().run_in_executor(None, client.connect)
        logger.info("🔵 Connected successfully - inventory should be running")
        logger.debug("Client attributes: %s", dir(client))
        tag_task = asyncio.create_task(process_tags())
    except Exception as e:
        logger.error(f"🔴 Failed to connect: {e}")
        raise
    
    yield
    
    if client:
        logger.info("🔴 Disconnecting...")
        # Flush the queue before cancelling
        while not tag_queue.empty():
            tags = tag_queue.get()
            logger.debug(f"Shutdown - Dequeued tags: {tags}")
            for tag in tags:
                epc = tag.get('EPC', 'Unknown')
                antenna = tag.get('AntennaID', 'Unknown')
                rssi = tag.get('PeakRSSI', 'Unknown')
                phase_angle = tag.get('PhaseAngle', 'Unknown')
                frequency = tag.get('ChannelIndex', 'Unknown')
                read_count = tag.get('TagSeenCount', 0)
                doppler_frequency = tag.get('DopplerFrequency', 'Unknown')
                timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
                logger.info(
                    f"Shutdown - Timestamp: {timestamp}\n"
                    f"EPC: {epc}\n"
                    f"Antenna: {antenna}\n"
                    f"RSSI: {rssi}\n"
                    f"PhaseAngle: {phase_angle}\n"
                    f"Frequency: {frequency}\n"
                    f"ReadCount: {read_count}\n"
                    f"DopplerFrequency: {doppler_frequency}\n"
                )
        tag_task.cancel()
        try:
            await tag_task
        except asyncio.CancelledError:
            logger.debug("Tag processing task cancelled")
        await asyncio.get_event_loop().run_in_executor(None, client.disconnect)
        logger.info("🔴 Disconnected")

app = FastAPI(lifespan=lifespan)

@app.post("/rfid")
async def receive_llrp(request: Request):
    body = await request.body()
    logger.debug(f"🔻 Received POST data: {body}")
    return {"status": "LLRP data received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)