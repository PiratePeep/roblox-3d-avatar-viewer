import axios, { AxiosResponse, AxiosInstance } from "axios";

export interface LoadAvatarResponse {
  targetId: number;
  state: "Completed" | "Pending" | "Error" | "Blocked" | "InReview";
  imageUrl: string;
  version: string;
}

export interface SceneMetadataResponse {
  camera: {
    position: Max;
    direction: Max;
    fov: number;
  };
  aabb: {
    min: Max;
    max: Max;
  };
  mtl: string;
  obj: string;
  textures: string[];
}

export interface Max {
  x: number;
  y: number;
  z: number;
}

export interface UserResult {
  data: User[];
}

export interface User {
  requestedUsername: string;
  hasVerifiedBadge: boolean;
  id: number;
  name: string;
  displayName: string;
}

export class AvatarHelper {
  constructor() {}

  private session: AxiosInstance = axios.create({
    headers: {
      Origin: "https://www.roblox.com",
      Referer: "https://www.roblox.com/",
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
      "Accept-Encoding": "gzip, deflate, br, zstd",
    },
  });

  private getHashUrl(hash: string) {
    let st = 31;
    for (let ii = 0; ii < hash.length; ii++) {
      st ^= hash[ii].charCodeAt(0);
    }
    return `https://t${(st % 8).toString()}.rbxcdn.com/${hash}`;
  }

  private async getIdFromUsername(username: string): Promise<number> {
    const response: AxiosResponse<UserResult> = await this.session.post(
      "https://users.roblox.com/v1/usernames/users",
      {
        usernames: [username],
      }
    );

    if (response.data.data.length === 0) {
      throw new Error("User not found");
    }
    return response.data.data[0].id;
  }

  async loadAvatar(username: string): Promise<LoadAvatarResponse> {
    const userId = await this.getIdFromUsername(username);
    const response: AxiosResponse<LoadAvatarResponse> = await this.session.get(
      `https://thumbnails.roblox.com/v1/users/avatar-3d?userId=${userId}`
    );
    return response.data;
  }

  async getSceneMetadata(imageUrl: string): Promise<SceneMetadataResponse> {
    const response: AxiosResponse<SceneMetadataResponse> =
      await this.session.get(imageUrl);
    return response.data;
  }

  // Don't think I'll need this since the OBJLoader and MTLLoader can handle the data directly
  async loadSceneData(sceneMetadata: SceneMetadataResponse) {
    const mtlResponse = await this.session.get(
      this.getHashUrl(sceneMetadata.mtl)
    );

    const objResponse = await this.session.get(
      this.getHashUrl(sceneMetadata.obj)
    );
    const textures = [];

    for (let i = 0; i < sceneMetadata.textures.length; i++) {
      let res = await this.session.get(
        this.getHashUrl(sceneMetadata.textures[i]),
        {
          responseType: "arraybuffer",
        }
      );

      textures.push(res.data);
    }

    return {
      mtl: mtlResponse.data,
      obj: objResponse.data,
      textures,
    };
  }
}
