import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ParticipantFlow } from "@/pages/ParticipantFlow"
import { Admin } from "@/pages/Admin"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/admin" element={<Admin />} />
        <Route path="/*" element={<ParticipantFlow />} />
      </Routes>
    </BrowserRouter>
  )
}
