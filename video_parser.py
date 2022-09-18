import asyncio
import aiohttp
import aiolimiter
import pandas as pd
from src.config import CONFIGURATION  as cfg

async def parse(video_link: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        async with aiohttp.ClientSession() as http_session:
            try:
                async with http_session.get(video_link) as http_res:
                    if http_res.status == 200:
                        video_buffer = await http_res.read()

                        # Указать путь к папке сохранения
                        with open(cfg.DATA + cfg.VID_PTH + '/' + video_link.split('/')[-1], "wb") as f:
                            f.write(video_buffer)

            except Exception as ex:
                print(ex)


async def main(csv_path: str):
    df = pd.read_csv(csv_path)
    tasks = []

    semaphore = asyncio.Semaphore(10)

    for url in df.video_uuid.values.tolist():
        tasks.append(asyncio.create_task(parse(video_link=url, semaphore=semaphore)))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    csv_path = cfg.DATA + cfg.VID_CSV
    asyncio.run(main(csv_path=csv_path))