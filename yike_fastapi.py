# from fastapi import FastAPI, HTTPException, Depends
# from pydantic import BaseModel
# import subprocess
#
# app = FastAPI()
#
# class LipSyncRequest(BaseModel):
#     audio: str
#     image: str
#
# @app.post("/api/sync")
# def lip_sync_endpoint(request: LipSyncRequest):
#     audio = request.audio
#     image = request.image
#
#     dir = "result"  # You can adjust the directory structure here if needed
#     command = (
#         f"python inference.py --driven_audio {audio} --source_image {image} "
#         f"--enhancer gfpgan --result_dir {dir} --size 256 --preprocess crop"
#     )
#     subprocess.run(command, shell=True)
#
#     output = subprocess.check_output(f"ls {dir}", shell=True).decode().splitlines()
#     result_file = output[0] if output else None
#
#     if result_file:
#         return {"result_path": f"./SadTalker/{dir}/{result_file}"}
#     else:
#         raise HTTPException(status_code=500, detail="Lip sync processing failed")
#
# if __name__ == "__main__":
#     import nest_asyncio
#     from uvicorn import run
#
#     nest_asyncio.apply()
#     run(app, host="0.0.0.0", port=8080)
