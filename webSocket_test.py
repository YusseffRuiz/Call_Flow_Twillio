import asyncio
import websockets

URL = "wss://nonspontaneous-befuddledly-patrina.ngrok-free.dev/twilio/ws"

async def main():
    print("Connecting to:", URL)
    async with websockets.connect(URL) as ws:
        print("Connected OK!")
        await ws.send("hello")
        await asyncio.sleep(2)

asyncio.run(main())
