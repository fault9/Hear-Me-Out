import { createWavFile } from "@shared/lib/audio"

export async function mergeAudioTracks(userWav: Blob, pplxWav: Blob): Promise<Blob> {
  const ctx = new AudioContext()
  const [userBuf, ppBuf] = await Promise.all([
    ctx.decodeAudioData(await userWav.arrayBuffer()),
    ctx.decodeAudioData(await pplxWav.arrayBuffer()),
  ])
  const maxLen = Math.max(userBuf.length, ppBuf.length)
  const merged = new Float32Array(maxLen)
  const user = userBuf.getChannelData(0)
  const assistant = ppBuf.getChannelData(0)
  const gain = 0.75
  let peak = 0
  for (let i = 0; i < maxLen; i++) {
    const sample = (i < user.length ? user[i] * gain : 0)
      + (i < assistant.length ? assistant[i] * gain : 0)
    merged[i] = sample
    peak = Math.max(peak, Math.abs(sample))
  }
  if (peak > 0.98) {
    const scale = 0.98 / peak
    for (let i = 0; i < merged.length; i++) merged[i] *= scale
  }
  ctx.close()
  return createWavFile(merged, userBuf.sampleRate)
}
