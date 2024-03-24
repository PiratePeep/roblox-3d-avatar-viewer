"use client";

import React, { useEffect, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, Sphere } from "@react-three/drei";
import useSWR from "swr";
import { Avatar } from "@/components/Avatar";

function LoadingSphere() {
  return (
    <Sphere args={[4, 32, 32]} position={[0, 0, -5]}>
      <meshStandardMaterial color="grey" />
    </Sphere>
  );
}

export function MainCanvas() {
  const [input, setInput] = React.useState("Roblox");
  const [username, setUsername] = React.useState("Roblox");

  const { data, error, isLoading } = useSWR(
    `/avatar?username=${username}`,
    async (url) => {
      const res = await fetch(url);
      return res.json();
    }
  );

  // This is necessary because NextJS is dumb and tries to download the files on the server instead of the client
  const [hydrated, setHydrated] = React.useState(false);

  useEffect(() => {
    setHydrated(true);
  }, []);

  if (!hydrated) {
    return null;
  }

  return (
    <>
      <div className="h-screen w-screen flex flex-col justify-center items-center bg-white gap-4">
        <h1 className="text-4xl font-bold text-black">3D Avatar Viewer</h1>
        <div className="h-1/2 w-1/2">
          <Suspense fallback={null}>
            <Canvas className="rounded-3xl" shadows={true}>
              <Environment
                background={true}
                blur={0.6}
                files="/hdrs/puresky.hdr"
              />
              <ambientLight intensity={Math.PI / 15} color={0xffffff} />
              <spotLight
                position={[40, 100, -10]}
                penumbra={1}
                decay={10}
                intensity={0.2}
              />

              {data && !isLoading && <Avatar data={data} />}

              {(isLoading || error) && <LoadingSphere />}

              <OrbitControls />
            </Canvas>
          </Suspense>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            setUsername(input);
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            placeholder="Roblox"
            className="rounded-xl p-2 border border-gray-300 text-black placeholder:text-gray-500"
            onChange={(e) => setInput(e.target.value)}
          />

          <button
            onClick={() => {
              setUsername(input);
            }}
            className="rounded-xl py-2 px-4 bg-black text-white"
          >
            Load
          </button>
        </form>
      </div>
    </>
  );
}
