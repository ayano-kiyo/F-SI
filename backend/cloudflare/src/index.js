import { DurableObject } from "cloudflare:workers";

const PORT = 8000;
const HEALTH_PATH = "/api/health";
const INACTIVITY_TIMEOUT_MS = 10 * 60 * 1000;

export class SimulatorContainer extends DurableObject {
  starting;

  async fetch(request) {
    this.starting ??= this.startAndWaitForHealth().finally(() => {
      this.starting = undefined;
    });
    await this.starting;

    const url = new URL(request.url);
    url.protocol = "http:";
    url.host = "container";
    const forwarded = new Request(url, request);
    forwarded.headers.delete("host");
    return this.ctx.container.getTcpPort(PORT).fetch(forwarded);
  }

  async startAndWaitForHealth() {
    const container = this.ctx.container;
    if (!container.running) {
      container.start({
        image: container.images.base,
        instance: "standard-1",
        enableInternet: false,
        env: {
          SIMULATOR_CORS_ORIGINS: this.env.SIMULATOR_CORS_ORIGINS ?? "*",
        },
      });
    }
    await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);

    const port = container.getTcpPort(PORT);
    let lastError;
    for (let attempt = 0; attempt < 120; attempt += 1) {
      try {
        const response = await port.fetch(`http://container${HEALTH_PATH}`, {
          signal: AbortSignal.timeout(1000),
        });
        await response.body?.cancel();
        if (!response.ok) {
          throw new Error(`Health check returned ${response.status}`);
        }
        return;
      } catch (error) {
        lastError = error;
        await scheduler.wait(250);
      }
    }

    throw new Error("Simulator container did not become ready on port 8000", {
      cause: lastError,
    });
  }
}

export default {
  fetch(request, env) {
    return env.SIMULATOR.getByName("global-race").fetch(request);
  },
};
