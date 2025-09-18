import type React from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/Button";
import {
  Activity,
  Users,
  FileText,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Brain,
  Stethoscope,
} from "lucide-react";

export const Dashboard: React.FC = () => {
  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">
          Early Disease Detection Dashboard
        </h1>
        <p className="text-muted-foreground">
          Monitor patient health and AI-powered disease detection metrics
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card>
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium">Total Patients</h3>
            <Users className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">2,847</div>
          <p className="text-xs text-muted-foreground">+12% from last month</p>
        </Card>

        <Card>
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium">AI Screenings</h3>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">156</div>
          <p className="text-xs text-muted-foreground">+8% from last week</p>
        </Card>

        <Card>
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium">Early Detections</h3>
            <TrendingUp className="h-4 w-4 text-chart-4" />
          </div>
          <div className="text-2xl font-bold">23</div>
          <p className="text-xs text-muted-foreground">+15% from last month</p>
        </Card>

        <Card>
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium">Risk Assessments</h3>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">89</div>
          <p className="text-xs text-muted-foreground">+5% from last week</p>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Alerts */}
        <Card className="lg:col-span-2">
          <div className="mb-4">
            <h2 className="flex items-center space-x-2 text-lg font-semibold">
              <AlertTriangle className="h-5 w-5 text-secondary" />
              <span>AI Detection Alerts</span>
            </h2>
            <p className="text-sm text-muted-foreground">
              Latest high-priority patient alerts from AI analysis
            </p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center space-x-4 p-3 bg-muted rounded-lg">
              <div className="w-2 h-2 bg-destructive rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm font-medium">High Risk: Patient #2847</p>
                <p className="text-xs text-muted-foreground">
                  AI detected cardiovascular risk factors - 94% confidence
                </p>
              </div>
              <span className="text-xs text-muted-foreground">2 min ago</span>
            </div>

            <div className="flex items-center space-x-4 p-3 bg-muted rounded-lg">
              <div className="w-2 h-2 bg-secondary rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm font-medium">
                  Medium Risk: Patient #2834
                </p>
                <p className="text-xs text-muted-foreground">
                  Diabetes markers detected - requires follow-up
                </p>
              </div>
              <span className="text-xs text-muted-foreground">15 min ago</span>
            </div>

            <div className="flex items-center space-x-4 p-3 bg-muted rounded-lg">
              <div className="w-2 h-2 bg-chart-3 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm font-medium">Follow-up: Patient #2821</p>
                <p className="text-xs text-muted-foreground">
                  Cancer screening analysis complete - 87% normal
                </p>
              </div>
              <span className="text-xs text-muted-foreground">1 hour ago</span>
            </div>
          </div>
        </Card>

        {/* Quick Actions */}
        <Card>
          <div className="mb-4">
            <h2 className="text-lg font-semibold">Quick Actions</h2>
            <p className="text-sm text-muted-foreground">
              Common tasks and AI tools
            </p>
          </div>
          <div className="space-y-3">
            <Button
              className="w-full justify-start bg-transparent"
              variant="outline"
            >
              <Users className="mr-2 h-4 w-4" />
              Add New Patient
            </Button>
            <Button
              className="w-full justify-start bg-transparent"
              variant="outline"
            >
              <Brain className="mr-2 h-4 w-4" />
              Start AI Screening
            </Button>
            <Button
              className="w-full justify-start bg-transparent"
              variant="outline"
            >
              <Activity className="mr-2 h-4 w-4" />
              View Analytics
            </Button>
            <Button
              className="w-full justify-start bg-transparent"
              variant="outline"
            >
              <Stethoscope className="mr-2 h-4 w-4" />
              Generate Report
            </Button>
          </div>
        </Card>
      </div>

      {/* Disease Detection Summary */}
      <Card className="mt-6">
        <div className="mb-4">
          <h2 className="flex items-center space-x-2 text-lg font-semibold">
            <Brain className="h-5 w-5 text-primary" />
            <span>AI Detection Summary</span>
          </h2>
          <p className="text-sm text-muted-foreground">
            Latest AI-powered disease detection results
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-muted rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Cardiovascular</span>
              <span className="text-xs bg-destructive text-destructive-foreground px-2 py-1 rounded">
                High Risk: 8
              </span>
            </div>
            <div className="w-full bg-border rounded-full h-2">
              <div
                className="bg-destructive h-2 rounded-full"
                style={{ width: "32%" }}
              ></div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              32% of screenings show elevated risk
            </p>
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Diabetes</span>
              <span className="text-xs bg-secondary text-secondary-foreground px-2 py-1 rounded">
                Medium Risk: 12
              </span>
            </div>
            <div className="w-full bg-border rounded-full h-2">
              <div
                className="bg-secondary h-2 rounded-full"
                style={{ width: "48%" }}
              ></div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              48% showing pre-diabetic markers
            </p>
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Cancer Screening</span>
              <span className="text-xs bg-chart-4 text-white px-2 py-1 rounded">
                Normal: 67
              </span>
            </div>
            <div className="w-full bg-border rounded-full h-2">
              <div
                className="bg-chart-4 h-2 rounded-full"
                style={{ width: "89%" }}
              ></div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              89% of screenings within normal range
            </p>
          </div>
        </div>
      </Card>

      {/* Recent Activity */}
      <Card className="mt-6">
        <div className="mb-4">
          <h2 className="flex items-center space-x-2 text-lg font-semibold">
            <Clock className="h-5 w-5" />
            <span>Recent Activity</span>
          </h2>
          <p className="text-sm text-muted-foreground">
            Latest system activities and patient updates
          </p>
        </div>
        <div className="space-y-4">
          <div className="flex items-center space-x-4">
            <CheckCircle className="h-5 w-5 text-chart-4" />
            <div className="flex-1">
              <p className="text-sm font-medium">
                AI screening completed for Patient #2847
              </p>
              <p className="text-xs text-muted-foreground">
                Dr. Sarah Johnson • 10 minutes ago
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <Brain className="h-5 w-5 text-primary" />
            <div className="flex-1">
              <p className="text-sm font-medium">
                New risk assessment generated by AI
              </p>
              <p className="text-xs text-muted-foreground">
                System • 25 minutes ago
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <Users className="h-5 w-5 text-secondary" />
            <div className="flex-1">
              <p className="text-sm font-medium">
                Patient #2834 registered for screening
              </p>
              <p className="text-xs text-muted-foreground">
                Reception • 1 hour ago
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};
