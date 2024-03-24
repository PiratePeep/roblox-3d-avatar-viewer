import { AvatarHelper } from "@/lib/avatar";

const loader = new AvatarHelper();

//TODO: Add error handling
export async function GET(request: Request) {
  const username = new URL(request.url).searchParams.get("username");

  if (!username) {
    return new Response("No username provided", {
      status: 400,
    });
  }

  const avatar = await loader.loadAvatar(username);
  const sceneMetadata = await loader.getSceneMetadata(avatar.imageUrl);

  return new Response(JSON.stringify(sceneMetadata), {
    headers: {
      "Content-Type": "application/json",
    },
  });
}
