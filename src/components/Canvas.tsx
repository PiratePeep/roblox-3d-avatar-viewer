"use client";

import dynamic from "next/dynamic";
import { Box } from "@/components/Box";
import React, { useEffect, useRef, Suspense } from "react";
import { Canvas, useLoader, extend } from "@react-three/fiber";
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";
//import { MTLLoader } from "three/examples/jsm/loaders/MTLLoader.js";
//import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import * as THREE from "three";
import { OrbitControls } from "@react-three/drei";
import useSWR from "swr";
import { MTLLoader } from "@/lib/MTLLoader";
import { OBJLoader } from "@/lib/OBJLoader";
extend({ RGBELoader, MTLLoader, OBJLoader });

// I'm a dumbass and I could've just used https://github.com/pmndrs/drei#environment
function Environment() {
  const hdr = useLoader(
    RGBELoader,
    "http://localhost:3000/hdrs/clarens_night_02_4k.hdr",
    () => {},
    () => {}
  );

  return (
    <mesh>
      <sphereGeometry args={[300, 60, 60]} />
      <meshPhongMaterial map={hdr} side={THREE.BackSide} />
    </mesh>
  );
}

function Avatar({
  data,
}: {
  data: {
    mtl: string;
    obj: string;
  };
}) {
  const mtl = useLoader(MTLLoader, data.mtl);
  const obj = useLoader(
    OBJLoader,
    data.obj,
    (loader) => {
      loader.setMaterials(mtl);
    },
    () => {}
  );

  return (
    <primitive
      object={obj}
      scale={2}
      position={[0, -206, -5]}
      rotation={[0, Math.PI, 0]}
    />
  );
}

export function MainCanvas() {
  // This can be cleaned up a lot
  const [input, setInput] = React.useState("Roblox");
  const [username, setUsername] = React.useState("Roblox");

  // TODO: Add a loading spinner and handle errors
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
      <Suspense fallback={<div>Loading...</div>}>
        <div className="h-screen w-screen flex flex-col justify-center items-center bg-white gap-4">
          <h1 className="text-4xl font-bold text-black">3D Avatar Viewer</h1>
          <div className="h-1/2 w-1/2">
            <Canvas className="rounded-3xl" shadows={true}>
              <Environment />
              <ambientLight intensity={Math.PI / 2.5} color={0xffefde} />
              <spotLight
                position={[40, 100, -10]}
                angle={1.2}
                penumbra={1}
                decay={0}
                intensity={Math.PI}
              />
              {/*
  
        <pointLight
          position={[-10, -10, -10]}
          decay={0}
          intensity={Math.PI}
        /> */}

              {/* <Box position={[0, 0, 0]} scale={3} /> */}

              {data && !isLoading && <Avatar data={data} />}
              <OrbitControls />
            </Canvas>
          </div>

          <form onSubmit={(e) => {
            e.preventDefault();
            setUsername(input);
          }} className="flex gap-2">
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
      </Suspense>
    </>
  );
}
