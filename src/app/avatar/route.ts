import { AvatarHelper } from "@/lib/avatar";

const loader = new AvatarHelper();

export async function GET(request: Request) {
  try {
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
  } catch (error) {
    return new Response("An error occurred", {
      status: 500,
    });
  }
}
