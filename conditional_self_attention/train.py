import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

FILE_DIR = Path(__file__).resolve().parent

DATA_DIR = FILE_DIR / "mnist_data"
CHECKPOINT_DIR = FILE_DIR / "checkpoints"
SAMPLE_DIR = FILE_DIR / "samples"

BEST_MODEL_PATH = FILE_DIR / "best_model.pt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10

BATCH_SIZE = 128
EPOCHS = 100

LEARNING_RATE = 2e-4

# Original DDPM uses 1000 diffusion steps
TIMESTEPS = 1000

# U-Net depth
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 4)

NUM_RES_BLOCKS = 2

# MNIST resolutions:
# 28 -> 14 -> 7
# Attention at 14x14 and 7x7
ATTENTION_RESOLUTIONS = (14, 7)

NUM_HEADS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TIME EMBEDDING
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):

        half_dim = self.dim // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=t.device
            )
            / (half_dim - 1)
        )

        embeddings = (
            t[:, None].float()
            * frequencies[None, :]
        )

        embeddings = torch.cat(
            [
                torch.sin(embeddings),
                torch.cos(embeddings)
            ],
            dim=1
        )

        return embeddings


# ============================================================
# SELF ATTENTION
# ============================================================

class SelfAttention(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=4
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(
            32,
            channels
        )

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1
        )

        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )


    def forward(self, x):

        B, C, H, W = x.shape

        residual = x

        x = self.norm(x)

        qkv = self.qkv(x)

        q, k, v = torch.chunk(
            qkv,
            3,
            dim=1
        )

        head_dim = C // self.num_heads


        # ----------------------------------------------------
        # reshape
        #
        # B,C,H,W
        #
        # ->
        #
        # B,heads,HW,head_dim
        # ----------------------------------------------------

        q = q.reshape(
            B,
            self.num_heads,
            head_dim,
            H * W
        )

        q = q.permute(
            0,
            1,
            3,
            2
        )


        k = k.reshape(
            B,
            self.num_heads,
            head_dim,
            H * W
        )


        v = v.reshape(
            B,
            self.num_heads,
            head_dim,
            H * W
        )

        v = v.permute(
            0,
            1,
            3,
            2
        )


        # ----------------------------------------------------
        # Q K^T
        # ----------------------------------------------------

        attention = torch.matmul(
            q,
            k
        )

        attention = attention / math.sqrt(
            head_dim
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )


        # ----------------------------------------------------
        # Attention * V
        # ----------------------------------------------------

        out = torch.matmul(
            attention,
            v
        )


        # ----------------------------------------------------
        # back to image format
        # ----------------------------------------------------

        out = out.permute(
            0,
            1,
            3,
            2
        )

        out = out.reshape(
            B,
            C,
            H,
            W
        )

        out = self.proj(out)

        return residual + out


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        embedding_dim
    ):
        super().__init__()


        self.norm1 = nn.GroupNorm(
            32,
            in_channels
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )


        # timestep + class information
        self.embedding_layer = nn.Linear(
            embedding_dim,
            out_channels
        )


        self.norm2 = nn.GroupNorm(
            32,
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )


        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.skip = nn.Identity()


    def forward(
        self,
        x,
        embedding
    ):

        residual = self.skip(x)


        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        h = self.norm1(x)

        h = F.silu(h)

        h = self.conv1(h)


        # ----------------------------------------------------
        # Inject timestep + class information
        # ----------------------------------------------------

        emb = self.embedding_layer(
            F.silu(embedding)
        )

        emb = emb[:, :, None, None]

        h = h + emb


        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        h = self.norm2(h)

        h = F.silu(h)

        h = self.conv2(h)


        return h + residual


# ============================================================
# DOWNSAMPLE
# ============================================================

class Downsample(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1
        )


    def forward(self, x):

        return self.conv(x)


# ============================================================
# UPSAMPLE
# ============================================================

class Upsample(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )


    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest"
        )

        return self.conv(x)


# ============================================================
# CONDITIONAL DDPM U-NET
# ============================================================

class ConditionalUNet(nn.Module):

    def __init__(
        self,
        image_channels=1,
        num_classes=10,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(14, 7),
        num_heads=4
    ):
        super().__init__()


        # ====================================================
        # EMBEDDING DIMENSION
        # ====================================================

        embedding_dim = base_channels * 4


        # ====================================================
        # TIMESTEP EMBEDDING
        # ====================================================

        self.time_embedding = nn.Sequential(

            SinusoidalTimeEmbedding(
                base_channels
            ),

            nn.Linear(
                base_channels,
                embedding_dim
            ),

            nn.SiLU(),

            nn.Linear(
                embedding_dim,
                embedding_dim
            )
        )


        # ====================================================
        # CLASS EMBEDDING
        # ====================================================

        self.class_embedding = nn.Embedding(
            num_classes,
            embedding_dim
        )


        # ====================================================
        # INITIAL CONVOLUTION
        # ====================================================

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )


        # ====================================================
        # ENCODER
        # ====================================================

        self.down_blocks = nn.ModuleList()

        current_channels = base_channels

        current_resolution = IMAGE_SIZE

        skip_channels = []


        for level, multiplier in enumerate(
            channel_mults
        ):

            output_channels = (
                base_channels
                * multiplier
            )


            for _ in range(
                num_res_blocks
            ):

                use_attention = (
                    current_resolution
                    in attention_resolutions
                )


                block = nn.ModuleDict({

                    "res":
                    ResBlock(
                        current_channels,
                        output_channels,
                        embedding_dim
                    ),

                    "attention":
                    SelfAttention(
                        output_channels,
                        num_heads
                    )
                    if use_attention
                    else nn.Identity()
                })


                self.down_blocks.append(
                    block
                )


                current_channels = (
                    output_channels
                )

                skip_channels.append(
                    current_channels
                )


            # -----------------------------------------------
            # Downsample except last level
            # -----------------------------------------------

            if level != len(channel_mults) - 1:

                self.down_blocks.append(

                    nn.ModuleDict({

                        "downsample":
                        Downsample(
                            current_channels
                        )
                    })

                )

                current_resolution //= 2


        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.middle_res1 = ResBlock(
            current_channels,
            current_channels,
            embedding_dim
        )

        self.middle_attention = SelfAttention(
            current_channels,
            num_heads
        )

        self.middle_res2 = ResBlock(
            current_channels,
            current_channels,
            embedding_dim
        )


        # ====================================================
        # DECODER
        # ====================================================

        self.up_blocks = nn.ModuleList()

        reversed_skip_channels = list(
            reversed(skip_channels)
        )

        skip_index = 0


        for level in reversed(
            range(len(channel_mults))
        ):

            output_channels = (
                base_channels
                * channel_mults[level]
            )


            for _ in range(
                num_res_blocks
            ):

                skip_ch = (
                    reversed_skip_channels[
                        skip_index
                    ]
                )

                skip_index += 1


                use_attention = (
                    current_resolution
                    in attention_resolutions
                )


                block = nn.ModuleDict({

                    "res":
                    ResBlock(
                        current_channels
                        + skip_ch,
                        output_channels,
                        embedding_dim
                    ),

                    "attention":
                    SelfAttention(
                        output_channels,
                        num_heads
                    )
                    if use_attention
                    else nn.Identity()
                })


                self.up_blocks.append(
                    block
                )

                current_channels = (
                    output_channels
                )


            # -----------------------------------------------
            # Upsample except final level
            # -----------------------------------------------

            if level != 0:

                self.up_blocks.append(

                    nn.ModuleDict({

                        "upsample":
                        Upsample(
                            current_channels
                        )
                    })

                )

                current_resolution *= 2


        # ====================================================
        # OUTPUT
        # ====================================================

        self.output_norm = nn.GroupNorm(
            32,
            current_channels
        )

        self.output_conv = nn.Conv2d(
            current_channels,
            image_channels,
            kernel_size=3,
            padding=1
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        timestep,
        labels
    ):


        # ----------------------------------------------------
        # Time embedding
        # ----------------------------------------------------

        time_emb = self.time_embedding(
            timestep
        )


        # ----------------------------------------------------
        # Class embedding
        # ----------------------------------------------------

        class_emb = self.class_embedding(
            labels
        )


        # ----------------------------------------------------
        # Conditional information
        #
        # timestep + class
        # ----------------------------------------------------

        embedding = (
            time_emb
            +
            class_emb
        )


        # ----------------------------------------------------
        # Input convolution
        # ----------------------------------------------------

        x = self.input_conv(x)

        skips = []


        # ====================================================
        # ENCODER
        # ====================================================

        for block in self.down_blocks:


            if "res" in block:

                x = block["res"](
                    x,
                    embedding
                )

                x = block["attention"](x)

                skips.append(x)


            else:

                x = block[
                    "downsample"
                ](x)


        # ====================================================
        # BOTTLENECK
        # ====================================================

        x = self.middle_res1(
            x,
            embedding
        )

        x = self.middle_attention(x)

        x = self.middle_res2(
            x,
            embedding
        )


        # ====================================================
        # DECODER
        # ====================================================

        for block in self.up_blocks:


            if "res" in block:

                skip = skips.pop()

                x = torch.cat(
                    [x, skip],
                    dim=1
                )

                x = block["res"](
                    x,
                    embedding
                )

                x = block[
                    "attention"
                ](x)


            else:

                x = block[
                    "upsample"
                ](x)


        # ====================================================
        # PREDICT NOISE
        # ====================================================

        x = self.output_norm(x)

        x = F.silu(x)

        predicted_noise = (
            self.output_conv(x)
        )

        return predicted_noise


# ============================================================
# DDPM
# ============================================================

class Diffusion:

    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda"
    ):

        self.timesteps = timesteps

        self.device = device


        # ====================================================
        # LINEAR BETA SCHEDULE
        #
        # Original DDPM style
        # ====================================================

        self.beta = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=device
        )


        self.alpha = (
            1.0
            -
            self.beta
        )


        self.alpha_bar = torch.cumprod(
            self.alpha,
            dim=0
        )


        self.alpha_bar_previous = torch.cat(
            [
                torch.ones(
                    1,
                    device=device
                ),

                self.alpha_bar[:-1]
            ]
        )


        # ====================================================
        # POSTERIOR VARIANCE
        # ====================================================

        self.posterior_variance = (

            self.beta

            *

            (
                1
                -
                self.alpha_bar_previous
            )

            /

            (
                1
                -
                self.alpha_bar
            )
        )


    # ========================================================
    # FORWARD DIFFUSION
    # ========================================================

    def add_noise(
        self,
        x0,
        t
    ):

        noise = torch.randn_like(
            x0
        )


        alpha_bar = (
            self.alpha_bar[t]
            [:, None, None, None]
        )


        noisy_image = (

            torch.sqrt(
                alpha_bar
            )

            * x0

            +

            torch.sqrt(
                1 - alpha_bar
            )

            * noise
        )


        return noisy_image, noise


# ============================================================
# SAMPLE IMAGES DURING TRAINING
# ============================================================

@torch.no_grad()
def generate_samples(
    model,
    diffusion,
    epoch
):

    model.eval()


    # ========================================================
    # Generate one sample for every digit:
    #
    # 0 1 2 3 4 5 6 7 8 9
    # ========================================================

    labels = torch.arange(
        10,
        device=DEVICE
    )


    x = torch.randn(
        10,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )


    print(
        "\nGenerating sample digits..."
    )


    # ========================================================
    # REVERSE DIFFUSION
    #
    # x_1000
    # ↓
    # x_999
    # ↓
    # ...
    # ↓
    # x_0
    # ========================================================

    for i in tqdm(
        reversed(
            range(
                diffusion.timesteps
            )
        ),
        total=diffusion.timesteps,
        desc="Sampling"
    ):


        t = torch.full(
            (10,),
            i,
            device=DEVICE,
            dtype=torch.long
        )


        predicted_noise = model(
            x,
            t,
            labels
        )


        alpha = (
            diffusion.alpha[i]
        )

        alpha_bar = (
            diffusion.alpha_bar[i]
        )

        beta = (
            diffusion.beta[i]
        )


        # ====================================================
        # DDPM REVERSE MEAN
        # ====================================================

        mean = (

            1
            /
            torch.sqrt(alpha)

            *

            (

                x

                -

                (
                    beta
                    /
                    torch.sqrt(
                        1
                        -
                        alpha_bar
                    )
                )

                *

                predicted_noise

            )
        )


        # ====================================================
        # Add noise except final step
        # ====================================================

        if i > 0:

            noise = torch.randn_like(
                x
            )


            variance = (
                diffusion.posterior_variance[i]
            )


            x = (

                mean

                +

                torch.sqrt(
                    variance
                )

                *

                noise
            )


        else:

            x = mean


    # ========================================================
    # Convert [-1,1] -> [0,1]
    # ========================================================

    x = torch.clamp(
        x,
        -1,
        1
    )

    x = (
        x + 1
    ) / 2


    sample_path = (

        SAMPLE_DIR
        /
        f"sample_epoch_{epoch}.png"
    )


    save_image(
        x,
        sample_path,
        nrow=5
    )


    print(
        f"Sample saved → {sample_path}"
    )


    model.train()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    diffusion
):

    model.eval()

    total_loss = 0


    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        batch_size = (
            images.shape[0]
        )


        # Random timestep
        t = torch.randint(
            0,
            TIMESTEPS,
            (batch_size,),
            device=DEVICE
        )


        # Add noise
        noisy_images, real_noise = (
            diffusion.add_noise(
                images,
                t
            )
        )


        # Predict noise
        predicted_noise = model(
            noisy_images,
            t,
            labels
        )


        # DDPM loss
        loss = F.mse_loss(
            predicted_noise,
            real_noise
        )


        total_loss += (
            loss.item()
        )


    model.train()


    return (
        total_loss
        /
        len(loader)
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    print(
        "\n========================================"
    )

    print(
        " CONDITIONAL DDPM — MNIST"
    )

    print(
        "========================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"File directory: {FILE_DIR}"
    )


    # ========================================================
    # DATASET
    # ========================================================

    transform = transforms.Compose([

        transforms.ToTensor(),

        # MNIST:
        # [0,1]
        # ->
        # [-1,1]

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])


    full_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    #
    # 55,000 training
    # 5,000 validation
    # ========================================================

    train_dataset, val_dataset = (
        random_split(

            full_dataset,

            [
                55000,
                5000
            ],

            generator=torch.Generator()
            .manual_seed(SEED)
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = ConditionalUNet(

        image_channels=IMAGE_CHANNELS,

        num_classes=NUM_CLASSES,

        base_channels=BASE_CHANNELS,

        channel_mults=CHANNEL_MULTS,

        num_res_blocks=NUM_RES_BLOCKS,

        attention_resolutions=
        ATTENTION_RESOLUTIONS,

        num_heads=NUM_HEADS

    ).to(DEVICE)


    # ========================================================
    # DIFFUSION
    # ========================================================

    diffusion = Diffusion(
        timesteps=TIMESTEPS,
        device=DEVICE
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        f"Model parameters: "
        f"{parameters / 1_000_000:.2f}M"
    )


    print(
        "\nStarting training...\n"
    )


    # ========================================================
    # BEST VALIDATION LOSS
    # ========================================================

    best_val_loss = float(
        "inf"
    )


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_train_loss = 0


        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{EPOCHS}"
            )
        )


        # ====================================================
        # BATCH LOOP
        # ====================================================

        for images, labels in progress_bar:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            batch_size = (
                images.shape[0]
            )


            # =================================================
            # STEP 1
            #
            # Random timestep
            # =================================================

            t = torch.randint(
                0,
                TIMESTEPS,
                (batch_size,),
                device=DEVICE
            )


            # =================================================
            # STEP 2
            #
            # Add Gaussian noise
            # =================================================

            noisy_images, real_noise = (
                diffusion.add_noise(
                    images,
                    t
                )
            )


            # =================================================
            # STEP 3
            #
            # Network predicts the noise
            # =================================================

            predicted_noise = model(
                noisy_images,
                t,
                labels
            )


            # =================================================
            # STEP 4
            #
            # Original simple DDPM objective
            #
            # MSE(
            # predicted noise,
            # real noise
            # )
            # =================================================

            loss = F.mse_loss(
                predicted_noise,
                real_noise
            )


            # =================================================
            # STEP 5
            #
            # BACKPROPAGATION
            # =================================================

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            optimizer.step()


            total_train_loss += (
                loss.item()
            )


            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        # ====================================================
        # TRAIN LOSS
        # ====================================================

        train_loss = (

            total_train_loss
            /
            len(train_loader)

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        val_loss = validate(
            model,
            val_loader,
            diffusion
        )


        print(
            f"\nEpoch {epoch}"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.6f}"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )


            torch.save(

                {

                    "epoch":
                    epoch,

                    "model_state_dict":
                    model.state_dict(),

                    "optimizer_state_dict":
                    optimizer.state_dict(),

                    "train_loss":
                    train_loss,

                    "val_loss":
                    val_loss,

                    "best_val_loss":
                    best_val_loss,

                    "config":
                    {

                        "image_size":
                        IMAGE_SIZE,

                        "image_channels":
                        IMAGE_CHANNELS,

                        "num_classes":
                        NUM_CLASSES,

                        "timesteps":
                        TIMESTEPS,

                        "base_channels":
                        BASE_CHANNELS,

                        "channel_mults":
                        CHANNEL_MULTS,

                        "num_res_blocks":
                        NUM_RES_BLOCKS,

                        "attention_resolutions":
                        ATTENTION_RESOLUTIONS,

                        "num_heads":
                        NUM_HEADS

                    }
                },

                BEST_MODEL_PATH
            )


            print(
                "\n✓ NEW BEST MODEL SAVED"
            )

            print(
                f"  {BEST_MODEL_PATH}"
            )

            print(
                f"  Best val loss: "
                f"{best_val_loss:.6f}"
            )


        # ====================================================
        # EVERY 5 EPOCHS
        #
        # SAVE CHECKPOINT
        #
        # GENERATE SAMPLES
        # ====================================================

        if epoch % 5 == 0:


            checkpoint_path = (

                CHECKPOINT_DIR

                /

                f"checkpoint_epoch_{epoch}.pt"
            )


            torch.save(

                {

                    "epoch":
                    epoch,

                    "model_state_dict":
                    model.state_dict(),

                    "optimizer_state_dict":
                    optimizer.state_dict(),

                    "train_loss":
                    train_loss,

                    "val_loss":
                    val_loss,

                    "best_val_loss":
                    best_val_loss

                },

                checkpoint_path
            )


            print(
                "\n✓ CHECKPOINT SAVED"
            )

            print(
                f"  {checkpoint_path}"
            )


            # =================================================
            # Generate:
            #
            # 0 1 2 3 4
            # 5 6 7 8 9
            # =================================================

            generate_samples(
                model,
                diffusion,
                epoch
            )


        print(
            "\n----------------------------------------\n"
        )


    print(
        "\n========================================"
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nBest model:"
        f"\n{BEST_MODEL_PATH}"
    )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )


# ============================================================
# START TRAINING
# ============================================================

if __name__ == "__main__":

    train()
