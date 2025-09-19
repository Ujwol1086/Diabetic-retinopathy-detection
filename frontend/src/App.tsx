import React, { useState } from "react"
import { DiseaseDetection } from "./components/DiseaseDetection"
import { Navigation } from "./components/Navigation"
import { ScreeningCenter } from "./components/ScreeningCenter"
import "./App.css"

function App() {
  const [activeTab, setActiveTab] = useState("detection")

  const renderContent = () => {
    switch (activeTab) {
      case "detection":
        return <DiseaseDetection />
      case "screening":
        return <ScreeningCenter />
      default:
        return <DiseaseDetection />
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderContent()}
      </main>
    </div>
  )
}

export default App