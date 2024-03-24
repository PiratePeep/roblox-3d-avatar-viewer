import { AvatarHelper } from "@/lib/avatar";

const loader = new AvatarHelper();

// Testing function for the avatar helper
async function main() {
  const avatar = await loader.loadAvatar("Roblox");
  const sceneMetadata = await loader.getSceneMetadata(avatar.imageUrl);
  console.log(sceneMetadata);
}

(async () => {
  try {
    await main();
  } catch (error) {}
})();
