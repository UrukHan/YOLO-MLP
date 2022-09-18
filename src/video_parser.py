import asyncio
from aiolimiter import AsyncLimiter
from aiohttp import ClientSession
from asyncio import Semaphore, create_task, wait
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from loguru import logger

# 1 реквест каждые 0.25 сек, можно сделать больше что бы ускорить процесс
limiter = AsyncLimiter(1, 0.25)

# Фейк юзер агенты
software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
user_agent_rotator = UserAgent(
    software_names=software_names, operating_systems=operating_systems, limit=100
)
user_agents = user_agent_rotator.get_user_agents()
user_agent = user_agent_rotator.get_random_user_agent()
headers = {"User-Agent": user_agent}

# Достаем айдишники из датафрейма
logger.info("Getting ids from dataframe")

# Функция выкачивает видос и записывает буфер на с3 сторедж/записывает буффер в downloads_dir
async def get_video_buffer_by_id(
    video_id,
    semaphore: Semaphore,
    video_path,
):
    video_url = f"https://center.prod.ritm.site/{video_id}/fhd.mp4"
    async with ClientSession(headers=headers) as session:
        await semaphore.acquire()
        async with limiter:
            try:
                async with session.get(video_url) as r:
                    logger.info(f"status: {r.status}, video_id: {video_id}")
                    if not r.status == 404:
                        video_buffer = await r.read()

                        # Если нужно сохранять видосы
                        with open(f"{video_path}/{video_id}.mp4", "wb") as video:
                            video.write(video_buffer)
                            semaphore.release()

                    else:
                        logger.error("status:", r.status, "video_id: ", video_id)
                        semaphore.release()
            except Exception as e:
                logger.exception(e)
                semaphore.release()


# Создание листа тасков
async def run_tasks(video_ids, download_dir):
    tasks = []
    semaphore = Semaphore(value=10)

    # Можно указать срез видосов
    for video_id in video_ids:

        # Если загружать видосы в с3 сторедж
        tasks.append(
            create_task(
                get_video_buffer_by_id(
                    video_id=video_id, semaphore=semaphore,
                    # boto_client=boto_client
                    video_path = download_dir
                )
            )
        )

    # Обязательно подождать таски!
    await wait(tasks)

def parse(video_ids, download_dir):
    loop = asyncio.new_event_loop().run_until_complete(run_tasks(video_ids, download_dir))
    asyncio.set_event_loop(loop)

    #asyncio.get_event_loop().run_until_complete(run_tasks(video_ids, download_dir))













